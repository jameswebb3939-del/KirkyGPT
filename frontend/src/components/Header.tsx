interface HeaderProps {
  onToggleSidebar: () => void;
}

export default function Header({
  onToggleSidebar,
}: HeaderProps) {
  return (
    <header className="app-header">
      <div className="header-left">
        <button
          className="sidebar-toggle"
          onClick={onToggleSidebar}
          aria-label="Toggle sidebar"
        >
          ☰
        </button>

        <div className="app-logo">
          Kirk
        </div>
      </div>

      <h1 className="app-title">
        KirkGPT
      </h1>
    </header>
  );
}